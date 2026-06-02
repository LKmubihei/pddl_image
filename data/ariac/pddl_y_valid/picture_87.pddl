(define (problem picture_87)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_regulator red_regulator green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_regulator)
        (clear red_regulator)
        (clear green_regulator)
        (part_at blue_regulator table)
        (part_at green_regulator table)
        (part_at red_regulator table)
    )
    
    (:goal
        (and
            (part_at blue_regulator regulator_placement)
        )
    )
)
