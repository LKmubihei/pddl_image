(define (problem picture_51)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator red_sensor green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear red_sensor)
        (clear green_regulator)
        (part_at red_regulator table)
        (part_at red_sensor table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
        )
    )
)
