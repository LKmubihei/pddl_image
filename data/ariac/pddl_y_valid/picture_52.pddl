(define (problem picture_52)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_sensor green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_sensor)
        (clear green_regulator)
        (part_at red_pump table)
        (part_at blue_sensor table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)
