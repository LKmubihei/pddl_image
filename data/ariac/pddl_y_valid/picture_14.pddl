(define (problem picture_14)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_sensor green_sensor red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_sensor)
        (clear red_regulator)
        (clear green_sensor)
        (part_at red_regulator table)
        (part_at green_sensor table)
        (part_at red_sensor table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
        )
    )
)
